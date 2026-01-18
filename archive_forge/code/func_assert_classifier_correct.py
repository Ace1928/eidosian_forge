import pytest
from nltk import classify
def assert_classifier_correct(algorithm):
    try:
        classifier = classify.MaxentClassifier.train(TRAIN, algorithm, trace=0, max_iter=1000)
    except (LookupError, AttributeError) as e:
        pytest.skip(str(e))
    for (px, py), featureset in zip(RESULTS, TEST):
        pdist = classifier.prob_classify(featureset)
        assert abs(pdist.prob('x') - px) < 0.01, (pdist.prob('x'), px)
        assert abs(pdist.prob('y') - py) < 0.01, (pdist.prob('y'), py)