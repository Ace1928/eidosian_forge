import os
from parlai.core.opt import Opt
from parlai.core.teachers import ParlAIDialogTeacher
from parlai.tasks.style_gen.build import (
def get_personality_list_path(opt: Opt) -> str:
    """
    Return the path to a list of personalities in the Image-Chat train set.
    """
    build_personality_list(opt)
    return os.path.join(opt['datapath'], TASK_FOLDER_NAME, 'personality_list.txt')