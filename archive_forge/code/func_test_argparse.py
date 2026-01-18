import argparse
import fixtures  # type: ignore
import typing
import autopage
import autopage.argparse
def test_argparse(self, module: typing.Any=autopage.argparse, color: bool=True) -> None:
    self.pager.return_value.to_terminal.return_value = color
    parser = module.ArgumentParser()
    try:
        parser.parse_args(['foo', '--help'])
    except SystemExit as exit:
        self.assertIs(self.pager.return_value.exit_code.return_value, exit.code)
    self.pager.assert_called_once_with(None, allow_color=True, line_buffering=False, reset_on_exit=False)
    self.pager.return_value.__enter__.assert_called_once()
    self.stream.seek(0)
    self.assertEqual('\x1b' in self.stream.read(), color)