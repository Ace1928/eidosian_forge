from appearing. If you want to prevent the settings instance from appearing
from the same thread and the other coroutines are only executed when Kivy
import os
from inspect import getfile
from os.path import dirname, join, exists, sep, expanduser, isfile
from kivy.config import ConfigParser
from kivy.base import runTouchApp, async_runTouchApp, stopTouchApp
from kivy.compat import string_types
from kivy.factory import Factory
from kivy.logger import Logger
from kivy.event import EventDispatcher
from kivy.lang import Builder
from kivy.resources import resource_find
from kivy.utils import platform
from kivy.uix.widget import Widget
from kivy.properties import ObjectProperty, StringProperty
from kivy.setupconfig import USE_SDL2
def load_kv(self, filename=None):
    """This method is invoked the first time the app is being run if no
        widget tree has been constructed before for this app.
        This method then looks for a matching kv file in the same directory as
        the file that contains the application class.

        For example, say you have a file named main.py that contains::

            class ShowcaseApp(App):
                pass

        This method will search for a file named `showcase.kv` in
        the directory that contains main.py. The name of the kv file has to be
        the lowercase name of the class, without the 'App' postfix at the end
        if it exists.

        You can define rules and a root widget in your kv file::

            <ClassName>: # this is a rule
                ...

            ClassName: # this is a root widget
                ...

        There must be only one root widget. See the :doc:`api-kivy.lang`
        documentation for more information on how to create kv files. If your
        kv file contains a root widget, it will be used as self.root, the root
        widget for the application.

        .. note::

            This function is called from :meth:`run`, therefore, any widget
            whose styling is defined in this kv file and is created before
            :meth:`run` is called (e.g. in `__init__`), won't have its styling
            applied. Note that :meth:`build` is called after :attr:`load_kv`
            has been called.
        """
    if filename:
        filename = resource_find(filename)
    else:
        try:
            default_kv_directory = dirname(getfile(self.__class__))
            if default_kv_directory == '':
                default_kv_directory = '.'
        except TypeError:
            default_kv_directory = '.'
        kv_directory = self.kv_directory or default_kv_directory
        clsname = self.__class__.__name__.lower()
        if clsname.endswith('app') and (not isfile(join(kv_directory, '%s.kv' % clsname))):
            clsname = clsname[:-3]
        filename = join(kv_directory, '%s.kv' % clsname)
    Logger.debug('App: Loading kv <{0}>'.format(filename))
    rfilename = resource_find(filename)
    if rfilename is None or not exists(rfilename):
        Logger.debug('App: kv <%s> not found' % filename)
        return False
    root = Builder.load_file(rfilename)
    if root:
        self.root = root
    return True