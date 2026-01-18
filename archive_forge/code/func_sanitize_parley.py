import copy
from parlai.core.teachers import ParlAIDialogTeacher, FbDialogTeacher
def sanitize_parley(parley):
    """
    Separate memories from context, pull out response, split context/memories lists.
    """
    if '\n' in parley.context:
        snippets = parley.context.split('\n')
        text = snippets[-1]
        mems = snippets[:-1]
        parley.context = text
        parley.memories = mems
    parley.response = parley.response[0]
    assert isinstance(parley.candidates, list)
    assert isinstance(parley.memories, list)
    return parley