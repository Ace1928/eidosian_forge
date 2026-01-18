from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import sys
from antlr3 import runtime_version, runtime_version_str
from antlr3.compat import set, frozenset, reversed
from antlr3.constants import DEFAULT_CHANNEL, HIDDEN_CHANNEL, EOF, \
from antlr3.exceptions import RecognitionException, MismatchedTokenException, \
from antlr3.tokens import CommonToken, EOF_TOKEN, SKIP_TOKEN
import six
from six import unichr
def computeContextSensitiveRuleFOLLOW(self):
    """
        Compute the context-sensitive FOLLOW set for current rule.
        This is set of token types that can follow a specific rule
        reference given a specific call chain.  You get the set of
        viable tokens that can possibly come next (lookahead depth 1)
        given the current call chain.  Contrast this with the
        definition of plain FOLLOW for rule r:

         FOLLOW(r)={x | S=>*alpha r beta in G and x in FIRST(beta)}

        where x in T* and alpha, beta in V*; T is set of terminals and
        V is the set of terminals and nonterminals.  In other words,
        FOLLOW(r) is the set of all tokens that can possibly follow
        references to r in *any* sentential form (context).  At
        runtime, however, we know precisely which context applies as
        we have the call chain.  We may compute the exact (rather
        than covering superset) set of following tokens.

        For example, consider grammar:

        stat : ID '=' expr ';'      // FOLLOW(stat)=={EOF}
             | "return" expr '.'
             ;
        expr : atom ('+' atom)* ;   // FOLLOW(expr)=={';','.',')'}
        atom : INT                  // FOLLOW(atom)=={'+',')',';','.'}
             | '(' expr ')'
             ;

        The FOLLOW sets are all inclusive whereas context-sensitive
        FOLLOW sets are precisely what could follow a rule reference.
        For input input "i=(3);", here is the derivation:

        stat => ID '=' expr ';'
             => ID '=' atom ('+' atom)* ';'
             => ID '=' '(' expr ')' ('+' atom)* ';'
             => ID '=' '(' atom ')' ('+' atom)* ';'
             => ID '=' '(' INT ')' ('+' atom)* ';'
             => ID '=' '(' INT ')' ';'

        At the "3" token, you'd have a call chain of

          stat -> expr -> atom -> expr -> atom

        What can follow that specific nested ref to atom?  Exactly ')'
        as you can see by looking at the derivation of this specific
        input.  Contrast this with the FOLLOW(atom)={'+',')',';','.'}.

        You want the exact viable token set when recovering from a
        token mismatch.  Upon token mismatch, if LA(1) is member of
        the viable next token set, then you know there is most likely
        a missing token in the input stream.  "Insert" one by just not
        throwing an exception.
        """
    return self.combineFollows(True)