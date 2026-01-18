import os
import json
import random
import tempfile
import subprocess
from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
def render_convo(opt):
    opt.log()
    extension = validate_args(opt)
    input_file, output_file = (opt['intput'], opt['output'])
    height, width = (opt['height'], opt['width'])
    alt_speaker = input_file.split('/')[-1][:-6]
    dialogs = pre_process(input_file, opt['num_examples'], alt_speaker)
    if output_file is None:
        display_cli(dialogs, alt_speaker, 'human')
    else:
        html_str = gen_html(dialogs, height, width, 'Rendered HTML', alt_speaker, 'human', opt['user_icon'], opt['alt_icon'])
        if extension == 'html':
            file_handle = open(output_file, 'w')
            file_handle.write(html_str)
            file_handle.close()
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                fname = tmpdir + '/interim.html'
                file_handle = open(fname, 'w')
                file_handle.write(html_str)
                if extension == 'pdf':
                    cmd = f'{CHROME_PATH} --headless --crash-dumps-dir=/tmp--print-to-pdf="{output_file}" {fname}'
                else:
                    cmd = f'{CHROME_PATH} --headless --hide-scrollbars--crash-dumps-dir=/tmp --window-size={opt['width'] * 100},{opt['height'] * 100}--screenshot="{output_file}" {fname}'
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
                file_handle.close()