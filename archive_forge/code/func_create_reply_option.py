import json
import logging
import requests
import parlai.chat_service.utils.logging as log_utils
def create_reply_option(title, payload=''):
    """
    Create a quick reply option.

    Takes in display title and optionally extra custom data.
    """
    assert len(title) <= MAX_QUICK_REPLY_TITLE_CHARS, 'Quick reply title length {} greater than the max of {}'.format(len(title), MAX_QUICK_REPLY_TITLE_CHARS)
    assert len(payload) <= MAX_POSTBACK_CHARS, 'Payload length {} greater than the max of {}'.format(len(payload), MAX_POSTBACK_CHARS)
    return {'content_type': 'text', 'title': title, 'payload': payload}