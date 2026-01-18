from huggingface_hub import model_info
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
from accelerate import init_empty_weights
from accelerate.commands.utils import CustomArgumentParser
from accelerate.utils import (
def create_empty_model(model_name: str, library_name: str, trust_remote_code: bool=False, access_token: str=None):
    """
    Creates an empty model from its parent library on the `Hub` to calculate the overall memory consumption.

    Args:
        model_name (`str`):
            The model name on the Hub
        library_name (`str`):
            The library the model has an integration with, such as `transformers`. Will be used if `model_name` has no
            metadata on the Hub to determine the library.
        trust_remote_code (`bool`, `optional`, defaults to `False`):
            Whether or not to allow for custom models defined on the Hub in their own modeling files. This option
            should only be set to `True` for repositories you trust and in which you have read the code, as it will
            execute code present on the Hub on your local machine.
        access_token (`str`, `optional`, defaults to `None`):
            The access token to use to access private or gated models on the Hub. (for use on the Gradio app)

    Returns:
        `torch.nn.Module`: The torch model that has been initialized on the `meta` device.

    """
    model_info = verify_on_hub(model_name, access_token)
    if model_info == 'gated':
        raise GatedRepoError(f'Repo for model `{model_name}` is gated. You must be authenticated to access it. Please run `huggingface-cli login`.')
    elif model_info == 'repo':
        raise RepositoryNotFoundError(f'Repo for model `{model_name}` does not exist on the Hub. If you are trying to access a private repo, make sure you are authenticated via `huggingface-cli login` and have access.')
    if library_name is None:
        library_name = getattr(model_info, 'library_name', False)
        if not library_name:
            raise ValueError(f'Model `{model_name}` does not have any library metadata on the Hub, please manually pass in a `--library_name` to use (such as `transformers`)')
    if library_name == 'transformers':
        if not is_transformers_available():
            raise ImportError(f'To check `{model_name}`, `transformers` must be installed. Please install it via `pip install transformers`')
        print(f'Loading pretrained config for `{model_name}` from `transformers`...')
        if model_info.config is None:
            raise RuntimeError(f'Tried to load `{model_name}` with `transformers` but it does not have any metadata.')
        auto_map = model_info.config.get('auto_map', False)
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code, token=access_token)
        with init_empty_weights():
            constructor = AutoModel
            if isinstance(auto_map, dict):
                value = None
                for key in auto_map.keys():
                    if key.startswith('AutoModelFor'):
                        value = key
                        break
                if value is not None:
                    constructor = getattr(transformers, value)
            model = constructor.from_config(config, trust_remote_code=trust_remote_code)
    elif library_name == 'timm':
        if not is_timm_available():
            raise ImportError(f'To check `{model_name}`, `timm` must be installed. Please install it via `pip install timm`')
        print(f'Loading pretrained config for `{model_name}` from `timm`...')
        with init_empty_weights():
            model = timm.create_model(model_name, pretrained=False)
    else:
        raise ValueError(f'Library `{library_name}` is not supported yet, please open an issue on GitHub for us to add support.')
    return model