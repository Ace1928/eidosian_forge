from __future__ import absolute_import, division, print_function
import traceback
from os import getenv
from os.path import isfile
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def manage_issue(taiga_host, project_name, issue_subject, issue_priority, issue_status, issue_type, issue_severity, issue_description, issue_attachment, issue_attachment_description, issue_tags, state, check_mode=False):
    """
    Method that creates/deletes issues depending whether they exist and the state desired

    The credentials should be passed via environment variables:
        - TAIGA_TOKEN
        - TAIGA_USERNAME and TAIGA_PASSWORD

    Returns a tuple with these elements:
        - A boolean representing the success of the operation
        - A descriptive message
        - A dict with the issue attributes, in case of issue creation, otherwise empty dict
    """
    changed = False
    try:
        token = getenv('TAIGA_TOKEN')
        if token:
            api = TaigaAPI(host=taiga_host, token=token)
        else:
            api = TaigaAPI(host=taiga_host)
            username = getenv('TAIGA_USERNAME')
            password = getenv('TAIGA_PASSWORD')
            if not any([username, password]):
                return (False, changed, 'Missing credentials', {})
            api.auth(username=username, password=password)
        user_id = api.me().id
        project_list = list(filter(lambda x: x.name == project_name, api.projects.list(member=user_id)))
        if len(project_list) != 1:
            return (False, changed, 'Unable to find project %s' % project_name, {})
        project = project_list[0]
        project_id = project.id
        priority_list = list(filter(lambda x: x.name == issue_priority, api.priorities.list(project=project_id)))
        if len(priority_list) != 1:
            return (False, changed, 'Unable to find issue priority %s for project %s' % (issue_priority, project_name), {})
        priority_id = priority_list[0].id
        status_list = list(filter(lambda x: x.name == issue_status, api.issue_statuses.list(project=project_id)))
        if len(status_list) != 1:
            return (False, changed, 'Unable to find issue status %s for project %s' % (issue_status, project_name), {})
        status_id = status_list[0].id
        type_list = list(filter(lambda x: x.name == issue_type, project.list_issue_types()))
        if len(type_list) != 1:
            return (False, changed, 'Unable to find issue type %s for project %s' % (issue_type, project_name), {})
        type_id = type_list[0].id
        severity_list = list(filter(lambda x: x.name == issue_severity, project.list_severities()))
        if len(severity_list) != 1:
            return (False, changed, 'Unable to find severity %s for project %s' % (issue_severity, project_name), {})
        severity_id = severity_list[0].id
        issue = {'project': project_name, 'subject': issue_subject, 'priority': issue_priority, 'status': issue_status, 'type': issue_type, 'severity': issue_severity, 'description': issue_description, 'tags': issue_tags}
        matching_issue_list = list(filter(lambda x: x.subject == issue_subject and x.type == type_id, project.list_issues()))
        matching_issue_list_len = len(matching_issue_list)
        if matching_issue_list_len == 0:
            if state == 'present':
                changed = True
                if not check_mode:
                    new_issue = project.add_issue(issue_subject, priority_id, status_id, type_id, severity_id, tags=issue_tags, description=issue_description)
                    if issue_attachment:
                        new_issue.attach(issue_attachment, description=issue_attachment_description)
                        issue['attachment'] = issue_attachment
                        issue['attachment_description'] = issue_attachment_description
                return (True, changed, 'Issue created', issue)
            else:
                return (True, changed, 'Issue does not exist', {})
        elif matching_issue_list_len == 1:
            if state == 'absent':
                changed = True
                if not check_mode:
                    matching_issue_list[0].delete()
                return (True, changed, 'Issue deleted', {})
            else:
                return (True, changed, 'Issue already exists', {})
        else:
            return (False, changed, 'More than one issue with subject %s in project %s' % (issue_subject, project_name), {})
    except TaigaException as exc:
        msg = 'An exception happened: %s' % to_native(exc)
        return (False, changed, msg, {})