from matplotlib import pylab
from nltk.corpus import gutenberg
from nltk.text import Text
def plot_word_freq_dist(text):
    fd = text.vocab()
    samples = [item for item, _ in fd.most_common(50)]
    values = [fd[sample] for sample in samples]
    values = [sum(values[:i + 1]) * 100.0 / fd.N() for i in range(len(values))]
    pylab.title(text.name)
    pylab.xlabel('Samples')
    pylab.ylabel('Cumulative Percentage')
    pylab.plot(values)
    pylab.xticks(range(len(samples)), [str(s) for s in samples], rotation=90)
    pylab.show()