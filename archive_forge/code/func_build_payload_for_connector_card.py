from __future__ import (absolute_import, division, print_function)
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import fetch_url
from ansible.module_utils.common.dict_transformations import snake_dict_to_camel_dict
def build_payload_for_connector_card(module, summary=None, color=None, title=None, text=None, actions=None, sections=None):
    payload = dict()
    payload['@context'] = OFFICE_365_CARD_CONTEXT
    payload['@type'] = OFFICE_365_CARD_TYPE
    if summary is not None:
        payload['summary'] = summary
    if color is not None:
        payload['themeColor'] = color
    if title is not None:
        payload['title'] = title
    if text is not None:
        payload['text'] = text
    if actions:
        payload['potentialAction'] = build_actions(actions)
    if sections:
        payload['sections'] = build_sections(sections)
    payload = module.jsonify(payload)
    return payload