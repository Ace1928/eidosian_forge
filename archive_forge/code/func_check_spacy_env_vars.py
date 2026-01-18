import os
from wasabi import msg
def check_spacy_env_vars():
    if 'SPACY_CONFIG_OVERRIDES' in os.environ:
        msg.warn("You've set a `SPACY_CONFIG_OVERRIDES` environment variable, which is now deprecated. Weasel will not use it. You can use `WEASEL_CONFIG_OVERRIDES` instead.")
    if 'SPACY_PROJECT_USE_GIT_VERSION' in os.environ:
        msg.warn("You've set a `SPACY_PROJECT_USE_GIT_VERSION` environment variable, which is now deprecated. Weasel will not use it.")