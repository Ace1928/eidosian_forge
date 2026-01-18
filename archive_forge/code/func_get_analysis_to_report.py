from __future__ import annotations
import sys
from typing import (
from coverage.exceptions import NoDataError, NotPython
from coverage.files import prep_patterns, GlobMatcher
from coverage.misc import ensure_dir_for_file, file_be_gone
from coverage.plugin import FileReporter
from coverage.results import Analysis
from coverage.types import TMorf
def get_analysis_to_report(coverage: Coverage, morfs: Iterable[TMorf] | None) -> Iterator[tuple[FileReporter, Analysis]]:
    """Get the files to report on.

    For each morf in `morfs`, if it should be reported on (based on the omit
    and include configuration options), yield a pair, the `FileReporter` and
    `Analysis` for the morf.

    """
    file_reporters = coverage._get_file_reporters(morfs)
    config = coverage.config
    if config.report_include:
        matcher = GlobMatcher(prep_patterns(config.report_include), 'report_include')
        file_reporters = [fr for fr in file_reporters if matcher.match(fr.filename)]
    if config.report_omit:
        matcher = GlobMatcher(prep_patterns(config.report_omit), 'report_omit')
        file_reporters = [fr for fr in file_reporters if not matcher.match(fr.filename)]
    if not file_reporters:
        raise NoDataError('No data to report.')
    for fr in sorted(file_reporters):
        try:
            analysis = coverage._analyze(fr)
        except NotPython:
            if fr.should_be_python():
                if config.ignore_errors:
                    msg = f"Couldn't parse Python file '{fr.filename}'"
                    coverage._warn(msg, slug='couldnt-parse')
                else:
                    raise
        except Exception as exc:
            if config.ignore_errors:
                msg = f"Couldn't parse '{fr.filename}': {exc}".rstrip()
                coverage._warn(msg, slug='couldnt-parse')
            else:
                raise
        else:
            yield (fr, analysis)