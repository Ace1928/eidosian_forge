from huggingface_hub import model_info
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from accelerate import init_empty_weights
from accelerate.commands.utils import CustomArgumentParser
from accelerate.utils import (
def check_has_model(error):
    """
    Checks what library spawned `error` when a model is not found
    """
    if is_timm_available() and isinstance(error, RuntimeError) and ('Unknown model' in error.args[0]):
        return 'timm'
    elif is_transformers_available() and isinstance(error, OSError) and ('does not appear to have a file named' in error.args[0]):
        return 'transformers'
    else:
        return 'unknown'