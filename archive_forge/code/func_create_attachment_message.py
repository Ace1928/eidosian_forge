import json
import logging
import requests
import parlai.chat_service.utils.logging as log_utils
def create_attachment_message(attachment_item, quick_replies=None):
    """
    Create a message list made with only an attachment.

    quick_replies should be a list of options made with create_reply_option.
    """
    payload = {'attachment': attachment_item}
    if quick_replies:
        assert len(quick_replies) <= MAX_QUICK_REPLIES, 'Number of quick replies {} greater than the max of {}'.format(len(quick_replies), MAX_QUICK_REPLIES)
        payload['quick_replies'] = quick_replies
    return [payload]