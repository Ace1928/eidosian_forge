import sys
import os
import shutil
import io
import re
import textwrap
from os.path import relpath
from errno import EEXIST
import traceback
def render_figures(code, code_path, output_dir, output_base, context, function_name, config, graph2use, simple_form, context_reset=False, close_figs=False):
    """
    Run a nipype workflow creation script and save the graph in *output_dir*.
    Save the images under *output_dir* with file names derived from
    *output_base*
    """
    formats = get_wf_formats(config)
    ns = wf_context if context else {}
    if context_reset:
        wf_context.clear()
    run_code(code, code_path, ns, function_name)
    img = ImageFile(output_base, output_dir)
    for fmt, dpi in formats:
        try:
            img_path = img.filename(fmt)
            imgname, ext = os.path.splitext(os.path.basename(img_path))
            ns['wf'].base_dir = output_dir
            src = ns['wf'].write_graph(imgname, format=ext[1:], graph2use=graph2use, simple_form=simple_form)
            shutil.move(src, img_path)
        except Exception:
            raise GraphError(traceback.format_exc())
        img.formats.append(fmt)
    return [(code, [img])]