from http.server import BaseHTTPRequestHandler, HTTPServer
from parlai.scripts.interactive import setup_args
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.image_featurizers import ImageLoader
from typing import Dict, Any
import json
import cgi
import PIL.Image as Image
from base64 import b64decode
import io
import os
def setup_interactive():
    """
    Set up the interactive script.
    """
    parser = setup_args()
    opt = parser.parse_args()
    if not opt.get('model_file'):
        raise RuntimeError('Please specify a model file')
    if opt.get('fixed_cands_path') is None:
        opt['fixed_cands_path'] = os.path.join('/'.join(opt.get('model_file').split('/')[:-1]), 'candidates.txt')
    opt['task'] = 'parlai.agents.local_human.local_human:LocalHumanAgent'
    opt['image_mode'] = 'resnet152'
    SHARED['opt'] = opt
    SHARED['image_loader'] = ImageLoader(opt)
    SHARED['agent'] = create_agent(opt, requireModelExists=True)
    SHARED['world'] = create_task(opt, SHARED['agent'])