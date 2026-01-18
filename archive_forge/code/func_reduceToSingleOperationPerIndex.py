from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import codecs
from antlr3.constants import DEFAULT_CHANNEL, EOF
from antlr3.tokens import Token, EOF_TOKEN
import six
from six import StringIO
def reduceToSingleOperationPerIndex(self, rewrites):
    """
        We need to combine operations and report invalid operations (like
        overlapping replaces that are not completed nested).  Inserts to
        same index need to be combined etc...   Here are the cases:

        I.i.u I.j.v                           leave alone, nonoverlapping
        I.i.u I.i.v                           combine: Iivu

        R.i-j.u R.x-y.v | i-j in x-y          delete first R
        R.i-j.u R.i-j.v                       delete first R
        R.i-j.u R.x-y.v | x-y in i-j          ERROR
        R.i-j.u R.x-y.v | boundaries overlap  ERROR

        I.i.u R.x-y.v   | i in x-y            delete I
        I.i.u R.x-y.v   | i not in x-y        leave alone, nonoverlapping
        R.x-y.v I.i.u   | i in x-y            ERROR
        R.x-y.v I.x.u                         R.x-y.uv (combine, delete I)
        R.x-y.v I.i.u   | i not in x-y        leave alone, nonoverlapping

        I.i.u = insert u before op @ index i
        R.x-y.u = replace x-y indexed tokens with u

        First we need to examine replaces.  For any replace op:

          1. wipe out any insertions before op within that range.
          2. Drop any replace op before that is contained completely within
             that range.
          3. Throw exception upon boundary overlap with any previous replace.

        Then we can deal with inserts:

          1. for any inserts to same index, combine even if not adjacent.
          2. for any prior replace with same left boundary, combine this
             insert with replace and delete this replace.
          3. throw exception if index in same range as previous replace

        Don't actually delete; make op null in list. Easier to walk list.
        Later we can throw as we add to index -> op map.

        Note that I.2 R.2-2 will wipe out I.2 even though, technically, the
        inserted stuff would be before the replace range.  But, if you
        add tokens in front of a method body '{' and then delete the method
        body, I think the stuff before the '{' you added should disappear too.

        Return a map from token index to operation.
        """
    for i, rop in enumerate(rewrites):
        if rop is None:
            continue
        if not isinstance(rop, ReplaceOp):
            continue
        for j, iop in self.getKindOfOps(rewrites, InsertBeforeOp, i):
            if iop.index >= rop.index and iop.index <= rop.lastIndex:
                rewrites[j] = None
        for j, prevRop in self.getKindOfOps(rewrites, ReplaceOp, i):
            if prevRop.index >= rop.index and prevRop.lastIndex <= rop.lastIndex:
                rewrites[j] = None
                continue
            disjoint = prevRop.lastIndex < rop.index or prevRop.index > rop.lastIndex
            same = prevRop.index == rop.index and prevRop.lastIndex == rop.lastIndex
            if not disjoint and (not same):
                raise ValueError('replace op boundaries of %s overlap with previous %s' % (rop, prevRop))
    for i, iop in enumerate(rewrites):
        if iop is None:
            continue
        if not isinstance(iop, InsertBeforeOp):
            continue
        for j, prevIop in self.getKindOfOps(rewrites, InsertBeforeOp, i):
            if prevIop.index == iop.index:
                iop.text = self.catOpText(iop.text, prevIop.text)
                rewrites[j] = None
        for j, rop in self.getKindOfOps(rewrites, ReplaceOp, i):
            if iop.index == rop.index:
                rop.text = self.catOpText(iop.text, rop.text)
                rewrites[i] = None
                continue
            if iop.index >= rop.index and iop.index <= rop.lastIndex:
                raise ValueError('insert op %s within boundaries of previous %s' % (iop, rop))
    m = {}
    for i, op in enumerate(rewrites):
        if op is None:
            continue
        assert op.index not in m, 'should only be one op per index'
        m[op.index] = op
    return m