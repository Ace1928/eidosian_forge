import os
import io
import sys
import zipfile
import shutil
from ipywidgets import embed as wembed
import ipyvolume
from ipyvolume.utils import download_to_file, download_to_bytes
from ipyvolume._version import __version_threejs__
def embed_html(filepath, widgets, makedirs=True, title=u'IPyVolume Widget', all_states=False, offline=False, scripts_path='js', drop_defaults=False, template=html_template, template_options=(('extra_script_head', ''), ('body_pre', ''), ('body_post', '')), devmode=False, offline_cors=False):
    """Write a minimal HTML file with widget views embedded.

    :param str filepath: The file to write the HTML output to.
    :type widgets: widget or collection of widgets or None
    :param widgets:The widgets to include views for. If None, all DOMWidgets are included (not just the displayed ones).
    :param makedirs: whether to make directories in the filename path, if they do not already exist
    :param title: title for the html page
    :param all_states: if True, the state of all widgets know to the widget manager is included, else only those in widgets
    :param offline: if True, use local urls for required js/css packages and download all js/css required packages
    (if not already available), such that the html can be viewed with no internet connection
    :param scripts_path: the directory to save required js/css packages to (relative to the filepath)
    :param bool drop_defaults: Whether to drop default values from the widget states
    :param template: template string for the html, must contain at least {title} and {snippet} place holders
    :param template_options: list or dict of additional template options
    :param devmode: if True, attempt to get index.js from local js/dist directory
    :param devmode: if True, attempt to get index.js from local js/dist folder
    :param offline_cors: if True, sets crossorigin attribute to anonymous, this allows for the return of error data
    from js scripts but can block local loading of the scripts in some browsers

    """
    dir_name_dst = os.path.dirname(os.path.abspath(filepath))
    if not os.path.exists(dir_name_dst) and makedirs:
        os.makedirs(dir_name_dst)
    template_opts = {'extra_script_head': '', 'body_pre': '', 'body_post': ''}
    template_opts.update(dict(template_options))
    if all_states:
        state = None
    else:
        state = wembed.dependency_state(widgets, drop_defaults=drop_defaults)
    if not offline:
        snippet = wembed.embed_snippet(widgets, state=state, requirejs=True, drop_defaults=drop_defaults)
    else:
        if not os.path.isabs(scripts_path):
            scripts_path = os.path.join(os.path.dirname(filepath), scripts_path)
        rel_script_path = os.path.relpath(scripts_path, os.path.dirname(filepath))
        if rel_script_path.startswith('..'):
            raise ValueError('The scripts_path must have the same root directory as the filepath')
        elif rel_script_path == '.':
            rel_script_path = ''
        else:
            rel_script_path += '/'
        fname_pyv = save_ipyvolumejs(scripts_path, devmode=devmode)
        fname_require = save_requirejs(os.path.join(scripts_path))
        fname_embed = save_embed_js(os.path.join(scripts_path))
        fname_fontawe = save_font_awesome(os.path.join(scripts_path))
        subsnippet = wembed.embed_snippet(widgets, embed_url=rel_script_path + fname_embed, requirejs=False, drop_defaults=drop_defaults, state=state)
        if not offline_cors:
            subsnippet = subsnippet.replace(' crossorigin="anonymous"', '')
        cors_attribute = 'crossorigin="anonymous"' if offline_cors else ' '
        snippet = '\n<link href="{rel_script_path}{fname_fontawe}/css/font-awesome.min.css" rel="stylesheet">\n<script src="{rel_script_path}{fname_require}"{cors} data-main=\'./{rel_script_path}\' ></script>\n<script>\n    require.config({{\n      map: {{\n        \'*\': {{\n          \'ipyvolume\': \'{fname_pyv}\',\n        }}\n      }}}})\n</script>\n{subsnippet}\n        '.format(rel_script_path=rel_script_path, fname_fontawe=fname_fontawe, fname_require=fname_require, fname_pyv=os.path.splitext(fname_pyv)[0], subsnippet=subsnippet, cors=cors_attribute)
    template_opts['snippet'] = snippet
    template_opts['title'] = title
    html_code = template.format(**template_opts)
    with io.open(filepath, 'w', encoding='utf8') as f:
        f.write(html_code)