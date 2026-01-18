import copy
from parlai.core.teachers import ParlAIDialogTeacher, FbDialogTeacher
def episode_to_examples(episode, histsz):
    """
    Converts an episode (list of Parleys) into self-feeding compatible examples.

    WARNING: we no longer require a histz when making a self-feeding file. Shortening of
    the history is typically done in the teacher file or in interactive mode.
    """
    examples = []
    history = []
    for parley in episode:
        history.append(parley.context)
        if histsz < 0:
            utterances = history
            context = add_person_tokens(utterances, last_speaker=1)
        elif histsz == 0:
            context = '__null__'
        else:
            utterances = history[-histsz:]
            context = add_person_tokens(utterances, last_speaker=1)
        example = Parley(context, parley.response, parley.reward, copy.deepcopy(parley.candidates))
        examples.append(example)
        history.append(parley.response)
    return examples