from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils.urls import fetch_url
def build_payload_for_slack(text, channel, thread_id, username, icon_url, icon_emoji, link_names, parse, color, attachments, blocks, message_id, prepend_hash):
    payload = {}
    if color == 'normal' and text is not None:
        payload = dict(text=escape_quotes(text))
    elif text is not None:
        payload = dict(attachments=[dict(text=escape_quotes(text), color=color, mrkdwn_in=['text'])])
    if channel is not None:
        if prepend_hash == 'auto':
            if channel.startswith(('#', '@', 'C0', 'GF', 'G0', 'CP')):
                payload['channel'] = channel
            else:
                payload['channel'] = '#' + channel
        elif prepend_hash == 'always':
            payload['channel'] = '#' + channel
        elif prepend_hash == 'never':
            payload['channel'] = channel
    if thread_id is not None:
        payload['thread_ts'] = thread_id
    if username is not None:
        payload['username'] = username
    if icon_emoji is not None:
        payload['icon_emoji'] = icon_emoji
    else:
        payload['icon_url'] = icon_url
    if link_names is not None:
        payload['link_names'] = link_names
    if parse is not None:
        payload['parse'] = parse
    if message_id is not None:
        payload['ts'] = message_id
    if attachments is not None:
        if 'attachments' not in payload:
            payload['attachments'] = []
    if attachments is not None:
        attachment_keys_to_escape = ['title', 'text', 'author_name', 'pretext', 'fallback']
        for attachment in attachments:
            for key in attachment_keys_to_escape:
                if key in attachment:
                    attachment[key] = escape_quotes(attachment[key])
            if 'fallback' not in attachment:
                attachment['fallback'] = attachment['text']
            payload['attachments'].append(attachment)
    if blocks is not None:
        block_keys_to_escape = ['text', 'alt_text']
        payload['blocks'] = recursive_escape_quotes(blocks, block_keys_to_escape)
    return payload