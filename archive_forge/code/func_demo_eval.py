import re
import regex
from nltk.chunk.api import ChunkParserI
from nltk.tree import Tree
def demo_eval(chunkparser, text):
    """
    Demonstration code for evaluating a chunk parser, using a
    ``ChunkScore``.  This function assumes that ``text`` contains one
    sentence per line, and that each sentence has the form expected by
    ``tree.chunk``.  It runs the given chunk parser on each sentence in
    the text, and scores the result.  It prints the final score
    (precision, recall, and f-measure); and reports the set of chunks
    that were missed and the set of chunks that were incorrect.  (At
    most 10 missing chunks and 10 incorrect chunks are reported).

    :param chunkparser: The chunkparser to be tested
    :type chunkparser: ChunkParserI
    :param text: The chunked tagged text that should be used for
        evaluation.
    :type text: str
    """
    from nltk import chunk
    from nltk.tree import Tree
    chunkscore = chunk.ChunkScore()
    for sentence in text.split('\n'):
        print(sentence)
        sentence = sentence.strip()
        if not sentence:
            continue
        gold = chunk.tagstr2tree(sentence)
        tokens = gold.leaves()
        test = chunkparser.parse(Tree('S', tokens), trace=1)
        chunkscore.score(gold, test)
        print()
    print('/' + '=' * 75 + '\\')
    print('Scoring', chunkparser)
    print('-' * 77)
    print('Precision: %5.1f%%' % (chunkscore.precision() * 100), ' ' * 4, end=' ')
    print('Recall: %5.1f%%' % (chunkscore.recall() * 100), ' ' * 6, end=' ')
    print('F-Measure: %5.1f%%' % (chunkscore.f_measure() * 100))
    if chunkscore.missed():
        print('Missed:')
        missed = chunkscore.missed()
        for chunk in missed[:10]:
            print('  ', ' '.join(map(str, chunk)))
        if len(chunkscore.missed()) > 10:
            print('  ...')
    if chunkscore.incorrect():
        print('Incorrect:')
        incorrect = chunkscore.incorrect()
        for chunk in incorrect[:10]:
            print('  ', ' '.join(map(str, chunk)))
        if len(chunkscore.incorrect()) > 10:
            print('  ...')
    print('\\' + '=' * 75 + '/')
    print()