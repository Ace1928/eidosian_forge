import json
import logging
import requests
import parlai.chat_service.utils.logging as log_utils
def create_list_element(element):
    assert 'title' in element, 'List elems must have a title'
    ret_elem = {'title': element['title'], 'subtitle': '', 'default_action': {'type': 'postback', 'title': element['title'], 'payload': element['title']}}
    if 'subtitle' in element:
        ret_elem['subtitle'] = element['subtitle']
    return ret_elem