import re
from pathlib import Path
from typing import List, Optional, Union
from huggingface_hub import HfApi, HfFolder, get_hf_file_metadata, hf_hub_url
def find_files_matching_pattern(model_name_or_path: Union[str, Path], pattern: str, glob_pattern: str='**/*', subfolder: str='', use_auth_token: Optional[Union[bool, str]]=None, revision: Optional[str]=None) -> List[Path]:
    """
    Scans either a model repo or a local directory to find filenames matching the pattern.

    Args:
        model_name_or_path (`Union[str, Path]`):
            The name of the model repo on the Hugging Face Hub or the path to a local directory.
        pattern (`str`):
            The pattern to use to look for files.
        glob_pattern (`str`, defaults to `"**/*"`):
            The pattern to use to list all the files that need to be checked.
        subfolder (`str`, defaults to `""`):
            In case the model files are located inside a subfolder of the model directory / repo on the Hugging
            Face Hub, you can specify the subfolder name here.
        use_auth_token (`Optional[bool, str]`, *optional*):
            The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
            when running `transformers-cli login` (stored in `~/.huggingface`).
        revision (`Optional[str]`, defaults to `None`):
            Revision is the specific model version to use. It can be a branch name, a tag name, or a commit id.

    Returns:
        `List[Path]`
    """
    model_path = Path(model_name_or_path) if isinstance(model_name_or_path, str) else model_name_or_path
    pattern = re.compile(f'{subfolder}/{pattern}' if subfolder != '' else pattern)
    if model_path.is_dir():
        path = model_path
        files = model_path.glob(glob_pattern)
        files = [p for p in files if re.search(pattern, str(p))]
    else:
        path = model_name_or_path
        if isinstance(use_auth_token, bool):
            token = HfFolder().get_token()
        else:
            token = use_auth_token
        repo_files = map(Path, HfApi().list_repo_files(model_name_or_path, revision=revision, token=token))
        if subfolder != '':
            path = f'{path}/{subfolder}'
        files = [Path(p) for p in repo_files if re.match(pattern, str(p))]
    return files