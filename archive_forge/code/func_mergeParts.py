from functools import reduce
from pyparsing import Word, OneOrMore, Optional, Literal, NotAny, MatchFirst
from pyparsing import Group, oneOf, Suppress, ZeroOrMore, Combine, FollowedBy
from pyparsing import srange, CharsNotIn, StringEnd, LineEnd, White, Regex
from pyparsing import nums, alphas, alphanums, ParseException, Forward
import types, sys, os, re, datetime
def mergeParts(parts, vids, staves, rOpt, is_grand=0):
    if not staves:
        return (parts, vids)
    partsnew, vidsnew = ([], [])
    for voice_ids in staves:
        pixs = []
        for vid in voice_ids:
            if vid in vids:
                pixs.append(vids.index(vid))
            else:
                info('score partname %s does not exist' % vid)
        if pixs:
            xparts = [parts[pix] for pix in pixs]
            if len(xparts) > 1:
                mergedpart = mergePartList(xparts, rOpt, is_grand)
            else:
                mergedpart = xparts[0]
            partsnew.append(mergedpart)
            vidsnew.append(vids[pixs[0]])
    return (partsnew, vidsnew)