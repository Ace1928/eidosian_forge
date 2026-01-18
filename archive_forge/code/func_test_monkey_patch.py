import argparse
import fixtures  # type: ignore
import typing
import autopage
import autopage.argparse
def test_monkey_patch(self, color: bool=True) -> None:
    patch = self.useFixture(fixtures.MockPatch('argparse._HelpAction'))
    autopage.argparse.monkey_patch()
    self.assertIsNot(patch.mock, argparse._HelpAction)
    self.test_argparse(argparse, color)