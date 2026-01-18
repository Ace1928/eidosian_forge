from os import makedirs, path
import shutil
import sphinx
def add_js_files(app, config):
    jquery_installed = getattr(app, '_sphinxcontrib_jquery_installed', False)
    if sphinx.version_info[:2] >= (6, 0) and (not jquery_installed):
        makedirs(path.join(app.outdir, '_static'), exist_ok=True)
        for filename, integrity in _FILES:
            if config.jquery_use_sri:
                app.add_js_file(filename, priority=100, integrity=integrity)
            else:
                app.add_js_file(filename, priority=100)
            shutil.copyfile(path.join(_ROOT_DIR, filename), path.join(app.outdir, '_static', filename))
        app._sphinxcontrib_jquery_installed = True