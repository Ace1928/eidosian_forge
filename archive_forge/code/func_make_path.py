import os
import parlai.core.params as params
import parlai.core.build_data as build_data
def make_path(opt, fname):
    return os.path.join(opt['datapath'], FOLDER_NAME, fname)