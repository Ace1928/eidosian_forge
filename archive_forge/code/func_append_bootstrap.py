import pyomo.environ as pyo
def append_bootstrap(self, bootstrap_theta):
    """Append a bootstrap theta df to the scenario set; equally likely

        Args:
            bootstrap_theta (dataframe): created by the bootstrap
        Note: this can be cleaned up a lot with the list becomes a df,
              which is why I put it in the ScenarioSet class.
        """
    assert len(bootstrap_theta) > 0
    prob = 1.0 / len(bootstrap_theta)
    dfdict = bootstrap_theta.to_dict(orient='index')
    for index, ThetaVals in dfdict.items():
        name = 'Bootstrap' + str(index)
        self.addone(ParmestScen(name, ThetaVals, prob))