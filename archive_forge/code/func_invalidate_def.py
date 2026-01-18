from mako import util
def invalidate_def(self, name):
    """Invalidate the cached content of a particular ``<%def>`` within this
        template.

        """
    self.invalidate('render_%s' % name, __M_defname='render_%s' % name)