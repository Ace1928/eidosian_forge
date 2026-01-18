import logging
import uuid
def censored_copy(data_dict, censor_keys):
    """Returns redacted dict copy for censored keys"""
    if censor_keys is None:
        censor_keys = []
    return {k: v if k not in censor_keys else '<redacted>' for k, v in data_dict.items()}