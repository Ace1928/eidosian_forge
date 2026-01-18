from __future__ import annotations
import contextlib
import dataclasses
import gzip
import logging
from typing import (
from torch.onnx._internal.diagnostics import infra
from torch.onnx._internal.diagnostics.infra import formatter, sarif, utils
from torch.onnx._internal.diagnostics.infra.sarif import version as sarif_version
@contextlib.contextmanager
def log_section(self, level: int, message: str, *args, **kwargs) -> Generator[None, None, None]:
    """
        Context manager for a section of log messages, denoted by a title message and increased indentation.

        Same api as `logging.Logger.log`.

        This context manager logs the given title at the specified log level, increases the current
        section depth for subsequent log messages, and ensures that the section depth is decreased
        again when exiting the context.

        Args:
            level: The log level.
            message: The title message to log.
            *args: The arguments to the message. Use `LazyString` to defer the
                expensive evaluation of the arguments until the message is actually logged.
            **kwargs: The keyword arguments for `logging.Logger.log`.

        Yields:
            None: This context manager does not yield any value.

        Example:
            >>> with DiagnosticContext("DummyContext", "1.0"):
            ...     rule = infra.Rule("RuleID", "DummyRule", "Rule message")
            ...     diagnostic = Diagnostic(rule, infra.Level.WARNING)
            ...     with diagnostic.log_section(logging.INFO, "My Section"):
            ...         diagnostic.log(logging.INFO, "My Message")
            ...         with diagnostic.log_section(logging.INFO, "My Subsection"):
            ...             diagnostic.log(logging.INFO, "My Submessage")
            ...     diagnostic.additional_messages
            ['## My Section', 'My Message', '### My Subsection', 'My Submessage']
        """
    if self.logger.isEnabledFor(level):
        indented_format_message = f'##{'#' * self._current_log_section_depth} {message}'
        self.log(level, indented_format_message, *args, **kwargs)
    self._current_log_section_depth += 1
    try:
        yield
    finally:
        self._current_log_section_depth -= 1