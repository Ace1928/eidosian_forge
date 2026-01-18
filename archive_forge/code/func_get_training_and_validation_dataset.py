import numpy as np
import tensorflow.compat.v2 as tf
from keras.src.utils import dataset_utils
from tensorflow.python.util.tf_export import keras_export
def get_training_and_validation_dataset(file_paths, labels, validation_split, directory, label_mode, class_names, sampling_rate, output_sequence_length, ragged):
    file_paths_train, labels_train = dataset_utils.get_training_or_validation_split(file_paths, labels, validation_split, 'training')
    if not file_paths_train:
        raise ValueError(f'No training audio files found in directory {directory}. Allowed format(s): {ALLOWED_FORMATS}')
    file_paths_val, labels_val = dataset_utils.get_training_or_validation_split(file_paths, labels, validation_split, 'validation')
    if not file_paths_val:
        raise ValueError(f'No validation audio files found in directory {directory}. Allowed format(s): {ALLOWED_FORMATS}')
    train_dataset = paths_and_labels_to_dataset(file_paths=file_paths_train, labels=labels_train, label_mode=label_mode, num_classes=len(class_names), sampling_rate=sampling_rate, output_sequence_length=output_sequence_length, ragged=ragged)
    val_dataset = paths_and_labels_to_dataset(file_paths=file_paths_val, labels=labels_val, label_mode=label_mode, num_classes=len(class_names), sampling_rate=sampling_rate, output_sequence_length=output_sequence_length, ragged=ragged)
    return (train_dataset, val_dataset)