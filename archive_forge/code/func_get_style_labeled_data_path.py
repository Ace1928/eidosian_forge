import os
from parlai.core.opt import Opt
from parlai.core.teachers import ParlAIDialogTeacher
from parlai.tasks.style_gen.build import (
def get_style_labeled_data_path(opt: Opt, base_task: str) -> str:
    """
    Return the filepath for the specified datatype of the specified base task, with an
    Image-Chat personality attached to each example.
    """
    build_style_labeled_datasets(opt)
    dt = opt['datatype'].split(':')[0]
    return os.path.join(get_style_labeled_data_folder(opt['datapath']), base_task, dt + '.txt')