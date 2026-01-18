import os
from typing import Any, Callable, Dict, List, Tuple, Type
from .. import conflicts, errors, option, osutils, tests, transform
from ..bzr import conflicts as bzr_conflicts
from ..workingtree import WorkingTree
from . import scenarios, script
def test_bug_430129(self):
    self.assertRaises(transform.MalformedTransform, self.run_script, '\n$ brz init trunk\n...\n$ cd trunk\n$ brz mkdir foo\n...\n$ brz commit -m \'Create trunk\' -q\n$ rm -r foo\n$ echo "Boo!" >foo\n$ brz commit -m \'foo is now a file\' -q\n$ brz branch -q . -r 1 ../branch -q\n$ cd ../branch\n$ echo "Boing" >foo/bar\n$ brz add -q foo/bar -q\n$ brz commit -m \'Add foo/bar\' -q\n$ brz merge ../trunk\n2>brz: ERROR: Tree transform is malformed [(\'unversioned executability\', \'new-1\')]\n')