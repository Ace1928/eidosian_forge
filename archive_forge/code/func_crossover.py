import numpy as np
import math
from Algorithm import Algorithm
from Constants import USER_SEED
def crossover(self, networkA, networkB):
    weightsA = networkA.weights.copy()
    weightsB = networkB.weights.copy()
    biasesA = networkA.biases.copy()
    biasesB = networkB.biases.copy()
    for i in range(len(self.weights)):
        length = len(self.weights[i])
        split = np.random.uniform(0, 1, size=length)
        split = np.random.randint(1, length)
        self.weights[i] = weightsA[i].copy()
        self.weights[i][split > 0.5] = weightsB[i][split > 0.5].copy()
    for i in range(len(self.biases)):
        length = len(self.biases[i])
        split = np.random.randint(1, length)
        self.biases[i] = biasesA[i].copy()
        self.biases[i][:split] = biasesB[i][:split].copy()