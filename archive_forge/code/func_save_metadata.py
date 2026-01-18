import datetime
import json
import os
import random
from parlai.utils.misc import AttrDict
import parlai.utils.logging as logging
@classmethod
def save_metadata(cls, datapath, opt, self_chat=False, speakers=None, **kwargs):
    """
        Dump conversation metadata to file.
        """
    metadata = {}
    metadata['date'] = str(datetime.datetime.now())
    metadata['opt'] = opt
    metadata['self_chat'] = self_chat
    metadata['speakers'] = speakers
    metadata['version'] = cls.version()
    for k, v in kwargs.items():
        metadata[k] = v
    metadata_path = cls._get_path(datapath)
    logging.info(f'Writing metadata to file {metadata_path}')
    with open(metadata_path, 'w') as f:
        f.write(json.dumps(metadata))