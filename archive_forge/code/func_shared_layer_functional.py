import collections
import keras.src as keras
def shared_layer_functional():
    """Shared layer in a functional model."""
    main_input = keras.Input(shape=(10,), dtype='int32', name='main_input')
    x = keras.layers.Embedding(output_dim=5, input_dim=4, input_length=10)(main_input)
    lstm_out = keras.layers.LSTM(3)(x)
    auxiliary_output = keras.layers.Dense(1, activation='sigmoid', name='aux_output')(lstm_out)
    auxiliary_input = keras.Input(shape=(5,), name='aux_input')
    x = keras.layers.concatenate([lstm_out, auxiliary_input])
    x = keras.layers.Dense(2, activation='relu')(x)
    main_output = keras.layers.Dense(1, activation='sigmoid', name='main_output')(x)
    model = keras.Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])
    return ModelFn(model, [(None, 10), (None, 5)], [(None, 1), (None, 1)])