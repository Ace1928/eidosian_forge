def print_topic(self, topicno, topn=10):
    """Get a single topic as a formatted string.

        Parameters
        ----------
        topicno : int
            Topic id.
        topn : int
            Number of words from topic that will be used.

        Returns
        -------
        str
            String representation of topic, like '-0.340 * "category" + 0.298 * "$M$" + 0.183 * "algebra" + ... '.

        """
    return ' + '.join(('%.3f*"%s"' % (v, k) for k, v in self.show_topic(topicno, topn)))