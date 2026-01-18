import functools
import re
import warnings
@classmethod
def parse_simple(cls, simple):
    match = cls.NPM_SPEC_BLOCK.match(simple)
    prefix, major_t, minor_t, patch_t, prerel, build = match.groups()
    prefix = cls.PREFIX_ALIASES.get(prefix, prefix)
    major = None if major_t in cls.EMPTY_VALUES else int(major_t)
    minor = None if minor_t in cls.EMPTY_VALUES else int(minor_t)
    patch = None if patch_t in cls.EMPTY_VALUES else int(patch_t)
    if build is not None and prefix not in [cls.PREFIX_EQ]:
        build = None
    if major is None:
        target = Version(major=0, minor=0, patch=0)
        if prefix not in [cls.PREFIX_EQ, cls.PREFIX_GTE]:
            raise ValueError('Invalid expression %r' % simple)
        prefix = cls.PREFIX_GTE
    elif minor is None:
        target = Version(major=major, minor=0, patch=0)
    elif patch is None:
        target = Version(major=major, minor=minor, patch=0)
    else:
        target = Version(major=major, minor=minor, patch=patch, prerelease=prerel.split('.') if prerel else (), build=build.split('.') if build else ())
    if (major is None or minor is None or patch is None) and (prerel or build):
        raise ValueError('Invalid NPM spec: %r' % simple)
    if prefix == cls.PREFIX_CARET:
        if target.major:
            high = target.truncate().next_major()
        elif target.minor:
            high = target.truncate().next_minor()
        elif minor is None:
            high = target.truncate().next_major()
        elif patch is None:
            high = target.truncate().next_minor()
        else:
            high = target.truncate().next_patch()
        return [cls.range(Range.OP_GTE, target), cls.range(Range.OP_LT, high)]
    elif prefix == cls.PREFIX_TILDE:
        assert major is not None
        if minor is None:
            high = target.next_major()
        else:
            high = target.next_minor()
        return [cls.range(Range.OP_GTE, target), cls.range(Range.OP_LT, high)]
    elif prefix == cls.PREFIX_EQ:
        if major is None:
            return [cls.range(Range.OP_GTE, target)]
        elif minor is None:
            return [cls.range(Range.OP_GTE, target), cls.range(Range.OP_LT, target.next_major())]
        elif patch is None:
            return [cls.range(Range.OP_GTE, target), cls.range(Range.OP_LT, target.next_minor())]
        else:
            return [cls.range(Range.OP_EQ, target)]
    elif prefix == cls.PREFIX_GT:
        assert major is not None
        if minor is None:
            return [cls.range(Range.OP_GTE, target.next_major())]
        elif patch is None:
            return [cls.range(Range.OP_GTE, target.next_minor())]
        else:
            return [cls.range(Range.OP_GT, target)]
    elif prefix == cls.PREFIX_GTE:
        return [cls.range(Range.OP_GTE, target)]
    elif prefix == cls.PREFIX_LT:
        assert major is not None
        return [cls.range(Range.OP_LT, target)]
    else:
        assert prefix == cls.PREFIX_LTE
        assert major is not None
        if minor is None:
            return [cls.range(Range.OP_LT, target.next_major())]
        elif patch is None:
            return [cls.range(Range.OP_LT, target.next_minor())]
        else:
            return [cls.range(Range.OP_LTE, target)]