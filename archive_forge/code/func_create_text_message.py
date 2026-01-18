import json
import logging
import requests
import parlai.chat_service.utils.logging as log_utils
def create_text_message(text, quick_replies=None):
    """
    Return a list of text messages from the given text.

    If the message is too long it is split into multiple messages. quick_replies should
    be a list of options made with create_reply_option.
    """

    def _message(text_content, replies):
        payload = {'text': text_content[:MAX_TEXT_CHARS]}
        if replies:
            payload['quick_replies'] = replies
        return payload
    tokens = [s[:MAX_TEXT_CHARS] for s in text.split(' ')]
    splits = []
    cutoff = 0
    curr_length = 0
    if quick_replies:
        assert len(quick_replies) <= MAX_QUICK_REPLIES, 'Number of quick replies {} greater than the max of {}'.format(len(quick_replies), MAX_QUICK_REPLIES)
    for i in range(len(tokens)):
        if tokens[i] == '[*SPLIT*]':
            if ' '.join(tokens[cutoff:i - 1]).strip() != '':
                splits.append(_message(' '.join(tokens[cutoff:i]), None))
                cutoff = i + 1
                curr_length = 0
        if curr_length + len(tokens[i]) > MAX_TEXT_CHARS:
            splits.append(_message(' '.join(tokens[cutoff:i]), None))
            cutoff = i
            curr_length = 0
        curr_length += len(tokens[i]) + 1
    if cutoff < len(tokens):
        splits.append(_message(' '.join(tokens[cutoff:]), quick_replies))
    return splits