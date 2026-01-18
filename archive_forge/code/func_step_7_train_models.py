import csv
import gzip
import json
import math
import optparse
import os
import pickle
import re
import sys
from pickle import Unpickler
import numpy as np
import requests
from pylab import *
from scipy import interp, stats
from sklearn import cross_validation, metrics, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (auc, make_scorer, precision_score, recall_score,
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, SDWriter
from rdkit.ML.Descriptors import MoleculeDescriptors
from the one dimensional weights.
def step_7_train_models(self):
    """train models according to trafficlight using sklearn.ensamble.RandomForestClassifier
        self.model contains up to 10 models afterwards, use save_model_info(type) to create csv or html
        containing data for each model"""
    title_line = ['#', 'accuracy', 'MCC', 'precision', 'recall', 'f1', 'auc', 'kappa', 'prevalence', 'bias', 'pickel-File']
    self.csv_text = [title_line]
    TL_list = []
    property_list_list = []
    directory = os.getcwd().split('/')[-2:]
    dir_string = ';'.join(directory)
    for cpd in self.sd_entries:
        property_list = []
        property_name_list = []
        prop_name = cpd.GetPropNames()
        for property in prop_name:
            if property not in ['TL', 'value']:
                try:
                    f = float(cpd.GetProp(property))
                    if math.isnan(f) or math.isinf(f):
                        print('invalid: %s' % property)
                except ValueError:
                    print('valerror: %s' % property)
                    continue
                property_list.append(f)
                property_name_list.append(property)
            elif property == 'TL':
                TL_list.append(int(cpd.GetProp(property)))
            else:
                print(property)
                pass
        property_list_list.append(property_list)
    dataDescrs_array = np.asarray(property_list_list)
    dataActs_array = np.array(TL_list)
    for randomseedcounter in range(1, 11):
        if self.verbous:
            print('################################')
            print('try to calculate seed %d' % randomseedcounter)
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(dataDescrs_array, dataActs_array, test_size=0.4, random_state=randomseedcounter)
        clf_RF = RandomForestClassifier(n_estimators=100, random_state=randomseedcounter)
        clf_RF = clf_RF.fit(X_train, y_train)
        cv_counter = 5
        scores = cross_validation.cross_val_score(clf_RF, X_test, y_test, cv=cv_counter, scoring='accuracy')
        accuracy_CV = round(scores.mean(), 3)
        accuracy_std_CV = round(scores.std(), 3)
        calcMCC = make_scorer(metrics.matthews_corrcoef, greater_is_better=True, needs_threshold=False)
        scores = cross_validation.cross_val_score(clf_RF, X_test, y_test, cv=cv_counter, scoring=calcMCC)
        MCC_CV = round(scores.mean(), 3)
        MCC_std_CV = round(scores.std(), 3)
        scores = cross_validation.cross_val_score(clf_RF, X_test, y_test, cv=cv_counter, scoring='f1')
        scores_rounded = [round(x, 3) for x in scores]
        f1_CV = round(scores.mean(), 3)
        f1_std_CV = round(scores.std(), 3)
        scores = cross_validation.cross_val_score(clf_RF, X_test, y_test, cv=cv_counter, scoring='precision')
        scores_rounded = [round(x, 3) for x in scores]
        precision_CV = round(scores.mean(), 3)
        precision_std_CV = round(scores.std(), 3)
        scores = cross_validation.cross_val_score(clf_RF, X_test, y_test, cv=cv_counter, scoring='recall')
        scores_rounded = [round(x, 3) for x in scores]
        recall_CV = round(scores.mean(), 3)
        recall_std_CV = round(scores.std(), 3)
        scores = cross_validation.cross_val_score(clf_RF, X_test, y_test, cv=cv_counter, scoring='roc_auc')
        scores_rounded = [round(x, 3) for x in scores]
        auc_CV = round(scores.mean(), 3)
        auc_std_CV = round(scores.std(), 3)
        y_predict = clf_RF.predict(X_test)
        conf_matrix = metrics.confusion_matrix(y_test, y_predict)
        coh_kappa = cohens_kappa(conf_matrix)
        kappa = round(coh_kappa['kappa'], 3)
        kappa_stdev = round(coh_kappa['std_kappa'], 3)
        tp = conf_matrix[0][0]
        tn = conf_matrix[1][1]
        fp = conf_matrix[1][0]
        fn = conf_matrix[0][1]
        n = tn + fp
        p = tp + fn
        kappa_prevalence = round(float(abs(tp - tn)) / float(n), 3)
        kappa_bias = round(float(abs(fp - fn)) / float(n), 3)
        if self.verbous:
            print('test:')
            print('\tpos\tneg')
            print('true\t%d\t%d' % (tp, tn))
            print('false\t%d\t%d' % (fp, fn))
            print(conf_matrix)
            print('\ntrain:')
            y_predict2 = clf_RF.predict(X_train)
            conf_matrix2 = metrics.confusion_matrix(y_train, y_predict2)
            tp2 = conf_matrix2[0][0]
            tn2 = conf_matrix2[1][1]
            fp2 = conf_matrix2[1][0]
            fn2 = conf_matrix2[0][1]
            print('\tpos\tneg')
            print('true\t%d\t%d' % (tp2, tn2))
            print('false\t%d\t%d' % (fp2, fn2))
            print(conf_matrix2)
        result_string_cut = [randomseedcounter, str(accuracy_CV) + '_' + str(accuracy_std_CV), str(MCC_CV) + '_' + str(MCC_std_CV), str(precision_CV) + '_' + str(precision_std_CV), str(recall_CV) + '_' + str(recall_std_CV), str(f1_CV) + '_' + str(f1_std_CV), str(auc_CV) + '_' + str(auc_std_CV), str(kappa) + '_' + str(kappa_stdev), kappa_prevalence, kappa_bias, 'model_file.pkl']
        self.model.append(clf_RF)
        self.csv_text.append(result_string_cut)
    return True if len(self.model) > 0 else False