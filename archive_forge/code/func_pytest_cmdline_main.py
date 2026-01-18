import dataclasses
from typing import AbstractSet
from typing import Collection
from typing import List
from typing import Optional
from typing import TYPE_CHECKING
from typing import Union
from .expression import Expression
from .expression import ParseError
from .structures import EMPTY_PARAMETERSET_OPTION
from .structures import get_empty_parameterset_mark
from .structures import Mark
from .structures import MARK_GEN
from .structures import MarkDecorator
from .structures import MarkGenerator
from .structures import ParameterSet
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config import UsageError
from _pytest.config.argparsing import Parser
from _pytest.stash import StashKey
@hookimpl(tryfirst=True)
def pytest_cmdline_main(config: Config) -> Optional[Union[int, ExitCode]]:
    import _pytest.config
    if config.option.markers:
        config._do_configure()
        tw = _pytest.config.create_terminal_writer(config)
        for line in config.getini('markers'):
            parts = line.split(':', 1)
            name = parts[0]
            rest = parts[1] if len(parts) == 2 else ''
            tw.write('@pytest.mark.%s:' % name, bold=True)
            tw.line(rest)
            tw.line()
        config._ensure_unconfigure()
        return 0
    return None