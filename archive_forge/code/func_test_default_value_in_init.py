import os
import sys
import tempfile
import textwrap
import shutil
import subprocess
import unittest
from traits.api import (
from traits.testing.optional_dependencies import requires_numpy
def test_default_value_in_init(self):

    class MyTraitType(TraitType):
        pass
    trait_type = MyTraitType(default_value=23)
    self.assertEqual(trait_type.get_default_value(), (DefaultValue.constant, 23))
    trait_type = MyTraitType(default_value=None)
    self.assertEqual(trait_type.get_default_value(), (DefaultValue.constant, None))
    trait_type = MyTraitType()
    self.assertEqual(trait_type.get_default_value(), (DefaultValue.constant, Undefined))
    trait_type = MyTraitType(default_value=NoDefaultSpecified)
    self.assertEqual(trait_type.get_default_value(), (DefaultValue.constant, Undefined))