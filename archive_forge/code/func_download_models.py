import importlib
import json
import time
import datetime
import os
import requests
import shutil
import hashlib
import tqdm
import math
import zipfile
import parlai.utils.logging as logging
def download_models(opt, fnames, model_folder, version='v1.0', path='aws', use_model_type=False):
    """
    Download models into the ParlAI model zoo from a url.

    :param fnames: list of filenames to download
    :param model_folder: models will be downloaded into models/model_folder/model_type
    :param path: url for downloading models; defaults to downloading from AWS
    :param use_model_type: whether models are categorized by type in AWS
    """
    model_type = opt.get('model_type', None)
    if model_type is not None:
        dpath = os.path.join(opt['datapath'], 'models', model_folder, model_type)
    else:
        dpath = os.path.join(opt['datapath'], 'models', model_folder)
    if not built(dpath, version):
        for fname in fnames:
            logging.info(f'building data: {dpath}/{fname}')
        if built(dpath):
            remove_dir(dpath)
        make_dir(dpath)
        for fname in fnames:
            if path == 'aws':
                url = 'http://parl.ai/downloads/_models/'
                url += model_folder + '/'
                if use_model_type:
                    url += model_type + '/'
                url += fname
            else:
                url = path + '/' + fname
            download(url, dpath, fname)
            if '.tgz' in fname or '.gz' in fname or '.zip' in fname:
                untar(dpath, fname)
        mark_done(dpath, version)