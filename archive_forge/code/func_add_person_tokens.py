import copy
from parlai.core.teachers import ParlAIDialogTeacher, FbDialogTeacher
def add_person_tokens(responses, first_speaker=None, last_speaker=1):
    """Converts a list of responses into a single tag-separated string
    Args:
        responses: list of responses (strings)
        first_speaker: either 1 or 2; the owner of the first response
        last_speaker: either 1 or 2; the owner of the last response
            NOTE: if first_speaker is provided, it overrides last_speaker
    Output:
        text: the concatenated text

    e.g.,
    responses = ["How are you?", "I'm doing fine!", "I'm glad to hear it!"]
    result = add_person_tokens(responses)
    result: "__p1__ How are you? __p2__ I'm doing fine! __p1__ I'm glad to
        hear it!"
    """
    if first_speaker is None:
        first_speaker = (last_speaker + len(responses)) % 2 + 1
    speaker = first_speaker
    text = ''
    for response in responses:
        tag = f'__p{speaker}__'
        text += ' ' + tag + ' ' + response
        speaker = 1 if speaker == 2 else 2
    return text.strip()