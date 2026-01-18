from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def getXmlScores(abc_string, skip=0, num=1, rOpt=False, bOpt=False, fOpt=False):
    return [fixDoctype(xml_doc) for xml_doc in getXmlDocs(abc_string, skip=0, num=1, rOpt=False, bOpt=False, fOpt=False)]