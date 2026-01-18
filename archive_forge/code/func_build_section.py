from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
def build_section(section):
    section_payload = dict()
    if 'title' in section:
        section_payload['title'] = section['title']
    if 'start_group' in section:
        section_payload['startGroup'] = section['start_group']
    if 'activity_image' in section:
        section_payload['activityImage'] = section['activity_image']
    if 'activity_title' in section:
        section_payload['activityTitle'] = section['activity_title']
    if 'activity_subtitle' in section:
        section_payload['activitySubtitle'] = section['activity_subtitle']
    if 'activity_text' in section:
        section_payload['activityText'] = section['activity_text']
    if 'hero_image' in section:
        section_payload['heroImage'] = section['hero_image']
    if 'text' in section:
        section_payload['text'] = section['text']
    if 'facts' in section:
        section_payload['facts'] = section['facts']
    if 'images' in section:
        section_payload['images'] = section['images']
    if 'actions' in section:
        section_payload['potentialAction'] = build_actions(section['actions'])
    return section_payload