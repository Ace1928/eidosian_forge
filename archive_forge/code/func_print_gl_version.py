import sys
from os import environ
def print_gl_version():
    backend = cgl_get_initialized_backend_name()
    Logger.info('GL: Backend used <{}>'.format(backend))
    version = glGetString(GL_VERSION)
    vendor = glGetString(GL_VENDOR)
    renderer = glGetString(GL_RENDERER)
    Logger.info('GL: OpenGL version <{0}>'.format(version))
    Logger.info('GL: OpenGL vendor <{0}>'.format(vendor))
    Logger.info('GL: OpenGL renderer <{0}>'.format(renderer))
    major, minor = gl_get_version()
    Logger.info('GL: OpenGL parsed version: %d, %d' % (major, minor))
    if (major, minor) < MIN_REQUIRED_GL_VERSION and backend != 'mock':
        if hasattr(sys, '_kivy_opengl_required_func'):
            sys._kivy_opengl_required_func(major, minor, version, vendor, renderer)
        else:
            msg = 'GL: Minimum required OpenGL version (2.0) NOT found!\n\nOpenGL version detected: {0}.{1}\n\nVersion: {2}\nVendor: {3}\nRenderer: {4}\n\nTry upgrading your graphics drivers and/or your graphics hardware in case of problems.\n\nThe application will leave now.'.format(major, minor, version, vendor, renderer)
            Logger.critical(msg)
            msgbox(msg)
    if platform != 'android':
        Logger.info('GL: Shading version <{0}>'.format(glGetString(GL_SHADING_LANGUAGE_VERSION)))
    Logger.info('GL: Texture max size <{0}>'.format(glGetIntegerv(GL_MAX_TEXTURE_SIZE)[0]))
    Logger.info('GL: Texture max units <{0}>'.format(glGetIntegerv(GL_MAX_TEXTURE_IMAGE_UNITS)[0]))