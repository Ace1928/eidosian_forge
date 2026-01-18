from huggingface_hub import model_info
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from accelerate import init_empty_weights
from accelerate.commands.utils import CustomArgumentParser
from accelerate.utils import (
def make_row(left_char, middle_char, right_char):
    return f'{left_char}{middle_char.join([in_between * n for n in column_widths])}{in_between * diff}{right_char}'