from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.general.plugins.module_utils.storage.hpe3par import hpe3par
def delete_cpg(client_obj, cpg_name):
    try:
        if client_obj.cpgExists(cpg_name):
            client_obj.deleteCPG(cpg_name)
        else:
            return (True, False, 'CPG does not exist')
    except exceptions.ClientException as e:
        return (False, False, 'CPG delete failed | %s' % e)
    return (True, True, 'Deleted CPG %s successfully.' % cpg_name)