import contextlib
import threading
import sys
import warnings
import unittest  # noqa: F401
from traits.api import (
from traits.util.async_trait_wait import wait_for_condition
@contextlib.contextmanager
def assertMultiTraitChanges(self, objects, traits_modified, traits_not_modified):
    """ Assert that traits on multiple objects do or do not change.

        This combines some of the functionality of `assertTraitChanges` and
        `assertTraitDoesNotChange`.

        Parameters
        ----------
        objects : list of HasTraits
            The HasTraits class instances whose traits will change.

        traits_modified : list of str
            The extended trait names of trait expected to change.

        traits_not_modified : list of str
            The extended trait names of traits not expected to change.

        """
    with contextlib.ExitStack() as exit_stack:
        cms = []
        for obj in objects:
            for trait in traits_modified:
                cms.append(exit_stack.enter_context(self.assertTraitChanges(obj, trait)))
            for trait in traits_not_modified:
                cms.append(exit_stack.enter_context(self.assertTraitDoesNotChange(obj, trait)))
        yield tuple(cms)