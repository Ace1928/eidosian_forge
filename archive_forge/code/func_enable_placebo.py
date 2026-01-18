import json
import os
import traceback
from ansible.module_utils._text import to_native
from ansible.module_utils.ansible_release import __version__
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.six import binary_type
from ansible.module_utils.six import text_type
from .common import get_collection_info
from .exceptions import AnsibleBotocoreError
from .retries import AWSRetry
def enable_placebo(session):
    """
    Helper to record or replay offline modules for testing purpose.
    """
    if '_ANSIBLE_PLACEBO_RECORD' in os.environ:
        import placebo
        existing_entries = os.listdir(os.environ['_ANSIBLE_PLACEBO_RECORD'])
        idx = len(existing_entries)
        data_path = f'{os.environ['_ANSIBLE_PLACEBO_RECORD']}/{idx}'
        os.mkdir(data_path)
        pill = placebo.attach(session, data_path=data_path)
        pill.record()
    if '_ANSIBLE_PLACEBO_REPLAY' in os.environ:
        import shutil
        import placebo
        existing_entries = sorted([int(i) for i in os.listdir(os.environ['_ANSIBLE_PLACEBO_REPLAY'])])
        idx = str(existing_entries[0])
        data_path = os.environ['_ANSIBLE_PLACEBO_REPLAY'] + '/' + idx
        try:
            shutil.rmtree('_tmp')
        except FileNotFoundError:
            pass
        shutil.move(data_path, '_tmp')
        if len(existing_entries) == 1:
            os.rmdir(os.environ['_ANSIBLE_PLACEBO_REPLAY'])
        pill = placebo.attach(session, data_path='_tmp')
        pill.playback()