import abc
from email import header
from email import charset as _charset
from email.utils import _has_surrogates
def handle_defect(self, obj, defect):
    """Based on policy, either raise defect or call register_defect.

            handle_defect(obj, defect)

        defect should be a Defect subclass, but in any case must be an
        Exception subclass.  obj is the object on which the defect should be
        registered if it is not raised.  If the raise_on_defect is True, the
        defect is raised as an error, otherwise the object and the defect are
        passed to register_defect.

        This method is intended to be called by parsers that discover defects.
        The email package parsers always call it with Defect instances.

        """
    if self.raise_on_defect:
        raise defect
    self.register_defect(obj, defect)