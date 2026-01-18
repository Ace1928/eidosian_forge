from mako import util
def invalidate_closure(self, name):
    """Invalidate a nested ``<%def>`` within this template.

        Caching of nested defs is a blunt tool as there is no
        management of scope -- nested defs that use cache tags
        need to have names unique of all other nested defs in the
        template, else their content will be overwritten by
        each other.

        """
    self.invalidate(name, __M_defname=name)