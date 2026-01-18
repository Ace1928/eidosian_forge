import argparse
import codecs
import srt
import logging
import sys
import itertools
import os
def compose_suggest_on_fail(subs, strict=True):
    try:
        return srt.compose(subs, strict=strict, eol=os.linesep, in_place=True)
    except srt.SRTParseError as thrown_exc:
        log.critical('Parsing failed, maybe you need to pass a different encoding with --encoding?')
        raise