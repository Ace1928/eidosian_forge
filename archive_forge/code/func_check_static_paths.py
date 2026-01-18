import os
import warnings
import builtins
import cherrypy
def check_static_paths(self):
    """Check Application config for incorrect static paths."""
    request = cherrypy.request
    for sn, app in cherrypy.tree.apps.items():
        if not isinstance(app, cherrypy.Application):
            continue
        request.app = app
        for section in app.config:
            request.get_resource(section + '/dummy.html')
            conf = request.config.get
            if conf('tools.staticdir.on', False):
                msg = ''
                root = conf('tools.staticdir.root')
                dir = conf('tools.staticdir.dir')
                if dir is None:
                    msg = 'tools.staticdir.dir is not set.'
                else:
                    fulldir = ''
                    if os.path.isabs(dir):
                        fulldir = dir
                        if root:
                            msg = 'dir is an absolute path, even though a root is provided.'
                            testdir = os.path.join(root, dir[1:])
                            if os.path.exists(testdir):
                                msg += '\nIf you meant to serve the filesystem folder at %r, remove the leading slash from dir.' % (testdir,)
                    elif not root:
                        msg = 'dir is a relative path and no root provided.'
                    else:
                        fulldir = os.path.join(root, dir)
                        if not os.path.isabs(fulldir):
                            msg = '%r is not an absolute path.' % (fulldir,)
                    if fulldir and (not os.path.exists(fulldir)):
                        if msg:
                            msg += '\n'
                        msg += '%r (root + dir) is not an existing filesystem path.' % fulldir
                if msg:
                    warnings.warn('%s\nsection: [%s]\nroot: %r\ndir: %r' % (msg, section, root, dir))